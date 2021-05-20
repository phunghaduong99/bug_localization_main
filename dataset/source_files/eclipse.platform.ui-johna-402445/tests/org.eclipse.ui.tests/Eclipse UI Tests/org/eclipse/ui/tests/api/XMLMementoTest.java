/*******************************************************************************
 * Copyright (c) 2005, 2006 IBM Corporation and others.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     IBM Corporation - initial API and implementation
 *******************************************************************************/
package org.eclipse.ui.tests.api;

import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.io.StringWriter;

import junit.framework.TestCase;

import org.eclipse.ui.IMemento;
import org.eclipse.ui.WorkbenchException;
import org.eclipse.ui.XMLMemento;

/**
 * Testing XMLMemento (see bug 93262). Emphasis is on ensuring that the 3.1
 * version behaves just like the 3.0.1 version.
 * 
 * @since 3.1
 * 
 */
public class XMLMementoTest extends TestCase {

	private static final String[] TEST_STRINGS = { "value",
			" value with spaces ", "value.with.many.dots",
			"value_with_underscores", "value<with<lessthan",
			"value>with>greaterthan", "value&with&ampersand",
			"value\"with\"quote", "value#with#hash", "",
			/*
			 * the following cases are for bug 93720
			 */
			"\nvalue\nwith\nnewlines\n", "\tvalue\twith\ttab\t",
			"\rvalue\rwith\rreturn\r", };

	/*
	 * Class under test for XMLMemento createReadRoot(Reader)
	 */
	public void testCreateReadRootReaderExceptionCases() {
		try {
			XMLMemento.createReadRoot(new StringReader("Invalid format"));
			fail("should throw WorkbenchException because of invalid format");
		} catch (WorkbenchException e) {
			// expected
		}
		try {
			XMLMemento.createReadRoot(new StringReader(
					"<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>"));
			fail("should throw WorkbenchException because there is no element");
		} catch (WorkbenchException e) {
			// expected
		}
		try {
			XMLMemento.createReadRoot(new Reader() {

				public void close() throws IOException {
					throw new IOException();
				}

				public int read(char[] arg0, int arg1, int arg2)
						throws IOException {
					throw new IOException();
				}
			});
			fail("should throw WorkbenchException because of IOException");
		} catch (WorkbenchException e) {
			// expected
		}
	}

	public void testCreateReadRootReader() throws WorkbenchException {
		XMLMemento memento = XMLMemento
				.createReadRoot(new StringReader(
						"<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?><simple>some text data</simple>"));
		assertEquals("some text data", memento.getTextData());
	}

	/*
	 * Class under test for XMLMemento createReadRoot(Reader, String)
	 */
	public void testCreateReadRootReaderString() {
		// TODO - I don't know how to test this. The method is not called by
		// anyone as of 2005/04/05.
	}

	public void testCreateWriteRoot() {
		String[] rootTypes = { "type", "type.with.dots",
				"type_with_underscores" };
		for (int i = 0; i < rootTypes.length; i++) {
			String type = rootTypes[i];
			XMLMemento memento = XMLMemento.createWriteRoot(type);
			assertNotNull(memento);
		}
	}

	public void testSpacesInRootAreIllegal() {
		try {
			XMLMemento.createWriteRoot("with space");
			fail("should fail");
		} catch (Exception e) {
			// expected
		}
	}

	public void testSpacesInKeysAreIllegal() {
		XMLMemento memento = XMLMemento.createWriteRoot("foo");
		try {
			memento.createChild("with space", "bar");
			fail("should fail");
		} catch (Exception e) {
			// expected
		}
		try {
			memento.putString("with space", "bar");
			fail("should fail");
		} catch (Exception e) {
			// expected
		}
	}

	public void testCopyChild() throws WorkbenchException, IOException {

		testPutAndGet(new MementoChecker() {

			public void prepareAndCheckBeforeSerialization(
					XMLMemento mementoToSerialize) {
				IMemento child = mementoToSerialize.createChild("c", "i");
				fillMemento(child);
				IMemento copiedChild = mementoToSerialize.copyChild(child);
				assertEquals("i", copiedChild.getID());
				checkMemento(copiedChild, true);
			}

			public void checkAfterDeserialization(XMLMemento deserializedMemento) {
				IMemento child = deserializedMemento.getChild("c");
				checkMemento(child, true);
				IMemento[] children = deserializedMemento.getChildren("c");
				assertEquals(2, children.length);
				assertEquals("i", children[0].getID());
				checkMemento(children[0], true);
				assertEquals("i", children[1].getID());
				checkMemento(children[1], true);
			}
		});
	}

	/**
	 * Helper method to fill a memento to be checked later by checkMemento.
	 * 
	 * @param memento
	 */
	private void fillMemento(IMemento memento) {
		memento.putFloat("floatKey", 0.4f);
		memento.putInteger("integerKey", 324765);
		memento.putString("stringKey", "a string");
		memento.putTextData("some text data");
		memento.createChild("child1");
		memento.createChild("child2", "child2id1");
		memento.createChild("child2", "child2id2");
	}

	/**
	 * Helper method to check if the values set by fillMemento are still there.
	 * The boolean parameter is needed because in some cases
	 * (IMememento#putMemento), the text data gets lost.
	 * 
	 * @param memento
	 * @param checkForTextData
	 */
	protected void checkMemento(IMemento memento, boolean checkForTextData) {
		assertEquals(0.4f, memento.getFloat("floatKey").floatValue(), 0.0f);
		assertEquals(324765, memento.getInteger("integerKey").intValue());
		assertEquals("a string", memento.getString("stringKey"));
		if (checkForTextData) {
			assertEquals("some text data", memento.getTextData());
		}
		IMemento child1 = memento.getChild("child1");
		assertNotNull(child1);
		IMemento child2 = memento.getChild("child2");
		assertNotNull(child2);
		assertEquals("child2id1", child2.getID());
		IMemento[] children = memento.getChildren("child2");
		assertNotNull(children);
		assertEquals(2, children.length);
		assertEquals("child2id1", children[0].getID());
		assertEquals("child2id2", children[1].getID());
	}

	public void testCreateAndGetChild() throws WorkbenchException, IOException {
		final String type1 = "type1";
		final String type2 = "type2";
		final String id = "id";

		testPutAndGet(new MementoChecker() {

			public void prepareAndCheckBeforeSerialization(
					XMLMemento mementoToSerialize) {
				// check that nothing is there yet
				assertEquals(null, mementoToSerialize.getChild(type1));
				assertEquals(null, mementoToSerialize.getChild(type2));

				// creation without ID
				IMemento child1 = mementoToSerialize.createChild(type1);
				assertNotNull(child1);
				assertNotNull(mementoToSerialize.getChild(type1));

				// creation with ID
				IMemento child2 = mementoToSerialize.createChild(type2, id);
				assertNotNull(child2);
				assertNotNull(mementoToSerialize.getChild(type2));
				assertEquals(id, child2.getID());
			}

			public void checkAfterDeserialization(XMLMemento deserializedMemento) {
				IMemento child1 = deserializedMemento.getChild(type1);
				assertNotNull(child1);
				IMemento child2 = deserializedMemento.getChild(type2);
				assertNotNull(child2);
				assertEquals(id, child2.getID());
			}
		});
	}

	public void testGetChildren() throws WorkbenchException, IOException {
		final String type = "type";
		final String id1 = "id";
		final String id2 = "id2";

		testPutAndGet(new MementoChecker() {

			public void prepareAndCheckBeforeSerialization(
					XMLMemento mementoToSerialize) {
				// check that nothing is there yet
				assertEquals(null, mementoToSerialize.getChild(type));

				IMemento child1 = mementoToSerialize.createChild(type, id1);
				assertNotNull(child1);
				assertNotNull(mementoToSerialize.getChild(type));
				assertEquals(id1, child1.getID());

				// second child with the same type
				IMemento child2 = mementoToSerialize.createChild(type, id2);
				assertNotNull(child2);
				assertEquals(2, mementoToSerialize.getChildren(type).length);
				assertEquals(id2, child2.getID());
			}

			public void checkAfterDeserialization(XMLMemento deserializedMemento) {
				IMemento[] children = deserializedMemento.getChildren(type);
				assertNotNull(children);
				assertEquals(2, children.length);

				// this checks that the order is maintained.
				// the spec does not promise this, but clients
				// may rely on the current implementation behaviour.
				assertEquals(id1, children[0].getID());
				assertEquals(id2, children[1].getID());
			}
		});
	}

	public void testGetID() throws WorkbenchException, IOException {
		final String type = "type";

		String[] ids = { "id", "", "id.with.many.dots", "id_with_underscores",
				"id<with<lessthan", "id>with>greaterthan", "id&with&ampersand",
				"id\"with\"quote", "id#with#hash" };

		for (int i = 0; i < ids.length; i++) {
			final String id = ids[i];

			testPutAndGet(new MementoChecker() {

				public void prepareAndCheckBeforeSerialization(
						XMLMemento mementoToSerialize) {
					assertEquals(null, mementoToSerialize.getChild(type));
					IMemento child = mementoToSerialize.createChild(type, id);
					assertEquals(id, child.getID());
				}

				public void checkAfterDeserialization(
						XMLMemento deserializedMemento) {
					IMemento child = deserializedMemento.getChild(type);
					assertNotNull(child);
					assertEquals(id, child.getID());
				}
			});
		}
	}

	public void testPutAndGetFloat() throws WorkbenchException, IOException {
		final String key = "key";

		final Float[] values = new Float[] { new Float(-3.1415), new Float(1),
				new Float(0), new Float(4554.45235),
				new Float(Float.MAX_VALUE), new Float(Float.MIN_VALUE),
				new Float(Float.NaN), new Float(Float.POSITIVE_INFINITY),
				new Float(Float.NEGATIVE_INFINITY) };

		for (int i = 0; i < values.length; i++) {
			final Float value = values[i];
			testPutAndGet(new MementoChecker() {

				public void prepareAndCheckBeforeSerialization(
						XMLMemento mementoToSerialize) {
					assertEquals(null, mementoToSerialize.getFloat(key));
					mementoToSerialize.putFloat(key, value.floatValue());
					assertEquals(value, mementoToSerialize.getFloat(key));
				}

				public void checkAfterDeserialization(
						XMLMemento deserializedMemento) {
					assertEquals(value, deserializedMemento.getFloat(key));
				}
			});
		}
	}

	public void testPutAndGetInteger() throws WorkbenchException, IOException {
		final String key = "key";

		Integer[] values = new Integer[] { new Integer(36254), new Integer(0),
				new Integer(1), new Integer(-36254),
				new Integer(Integer.MAX_VALUE), new Integer(Integer.MIN_VALUE) };

		for (int i = 0; i < values.length; i++) {
			final Integer value = values[i];

			testPutAndGet(new MementoChecker() {

				public void prepareAndCheckBeforeSerialization(
						XMLMemento mementoToSerialize) {
					assertEquals(null, mementoToSerialize.getInteger(key));
					mementoToSerialize.putInteger(key, value.intValue());
					assertEquals(value, mementoToSerialize.getInteger(key));
				}

				public void checkAfterDeserialization(
						XMLMemento deserializedMemento) {
					assertEquals(value, deserializedMemento.getInteger(key));
				}
			});
		}

	}

	public void testPutMemento() throws WorkbenchException, IOException {
		testPutAndGet(new MementoChecker() {

			public void prepareAndCheckBeforeSerialization(
					XMLMemento mementoToSerialize) {
				mementoToSerialize.putTextData("unchanged text data");
				mementoToSerialize.putString("neverlost", "retained value");

				IMemento aMemento = XMLMemento.createWriteRoot("foo");
				fillMemento(aMemento);

				// note that this does not copy the text data:
				mementoToSerialize.putMemento(aMemento);

				// do not check for text data:
				checkMemento(mementoToSerialize, false);

				assertEquals("unchanged text data", mementoToSerialize
						.getTextData());
				assertEquals("retained value", mementoToSerialize
						.getString("neverlost"));
			}

			public void checkAfterDeserialization(XMLMemento deserializedMemento) {
				// do not check for text data:
				checkMemento(deserializedMemento, false);

				assertEquals("unchanged text data", deserializedMemento
						.getTextData());
				assertEquals("retained value", deserializedMemento
						.getString("neverlost"));
			}
		});
	}

	public void testPutAndGetString() throws IOException, WorkbenchException {
		final String key = "key";

		// values with newline, tab, or return characters lead to bug 93720.
		String[] values = TEST_STRINGS;

		for (int i = 0; i < values.length; i++) {
			final String value = values[i];

			testPutAndGet(new MementoChecker() {

				public void prepareAndCheckBeforeSerialization(
						XMLMemento mementoToSerialize) {
					assertEquals(null, mementoToSerialize.getString(key));
					String helper = value;
					mementoToSerialize.putString(key, value);
					assertEquals(value, mementoToSerialize.getString(key));
					helper.toString();
				}

				public void checkAfterDeserialization(
						XMLMemento deserializedMemento) {
					assertEquals(value, deserializedMemento.getString(key));
				}
			});
		}
	}

	public void testPutAndGetTextData() throws WorkbenchException, IOException {
		String[] values = TEST_STRINGS;

		for (int i = 0; i < values.length; i++) {
			final String data = values[i];
			testPutAndGet(new MementoChecker() {

				public void prepareAndCheckBeforeSerialization(
						XMLMemento mementoToSerialize) {
					assertEquals(null, mementoToSerialize.getTextData());
					mementoToSerialize.putTextData(data);
					assertEquals(data, mementoToSerialize.getTextData());
				}

				public void checkAfterDeserialization(
						XMLMemento deserializedMemento) {
					if (data.equals("")) {
						// this comes back as null...
						assertEquals(null, deserializedMemento.getTextData());
					} else {
						assertEquals(data, deserializedMemento.getTextData());
					}
				}
			});
		}
	}

	public void testLegalKeys() throws WorkbenchException, IOException {
		String[] legalKeys = { "value", "value.with.many.dots",
				"value_with_underscores" };

		for (int i = 0; i < legalKeys.length; i++) {
			final String key = legalKeys[i];
			testPutAndGet(new MementoChecker() {

				public void prepareAndCheckBeforeSerialization(
						XMLMemento mementoToSerialize) {
					assertEquals(null, mementoToSerialize.getString(key));
					try {
						mementoToSerialize.putString(key, "some string");
					} catch (RuntimeException ex) {
						System.out.println("offending key: '" + key + "'");
						throw ex;
					}
					assertEquals("some string", mementoToSerialize
							.getString(key));
				}

				public void checkAfterDeserialization(
						XMLMemento deserializedMemento) {
					assertEquals("some string", deserializedMemento
							.getString(key));
				}
			});
		}

	}

	public void testIllegalKeys() {
		String[] illegalKeys = { "", " ", " key", "key ", "key key", "\t",
				"\tkey", "key\t", "key\tkey", "\n", "\nkey", "key\n",
				"key\nkey", "key<with<lessthan", "key>with>greaterthan",
				"key&with&ampersand", "key#with#hash", "key\"with\"quote", "\"" };

		for (int i = 0; i < illegalKeys.length; i++) {
			final String key = illegalKeys[i];
			XMLMemento memento = XMLMemento.createWriteRoot("foo");
			try {
				memento.putString(key, "some string");
				fail("putString with illegal key should fail");
			} catch (Exception ex) {
				// expected
			}
		}
	}

	public void testPutTextDataWithChildrenBug93718()
			throws WorkbenchException, IOException {
		final String textData = "\n\tThis is\ntext data\n\t\twith newlines and \ttabs\t\n\t ";
		testPutAndGet(new MementoChecker() {

			public void prepareAndCheckBeforeSerialization(
					XMLMemento mementoToSerialize) {
				mementoToSerialize.createChild("type", "id");
				mementoToSerialize.putTextData(textData);
				mementoToSerialize.createChild("type", "id");
				mementoToSerialize.createChild("type", "id");
				assertEquals(textData, mementoToSerialize.getTextData());
			}

			public void checkAfterDeserialization(XMLMemento deserializedMemento) {
				assertEquals(textData, deserializedMemento.getTextData());
			}
		});
	}

	private static interface MementoChecker {
		void prepareAndCheckBeforeSerialization(XMLMemento mementoToSerialize);

		void checkAfterDeserialization(XMLMemento deserializedMemento);
	}

	private void testPutAndGet(MementoChecker mementoChecker)
			throws IOException, WorkbenchException {
		XMLMemento mementoToSerialize = XMLMemento
				.createWriteRoot("XMLMementoTest");

		mementoChecker.prepareAndCheckBeforeSerialization(mementoToSerialize);

		StringWriter writer = new StringWriter();
		mementoToSerialize.save(writer);
		writer.close();

		StringReader reader = new StringReader(writer.getBuffer().toString());
		XMLMemento deserializedMemento = XMLMemento.createReadRoot(reader);

		mementoChecker.checkAfterDeserialization(deserializedMemento);
	}
	
	   public void testMementoWithTextContent113659() throws Exception {
	        IMemento memento = XMLMemento.createWriteRoot("root");
	        IMemento mementoWithChild = XMLMemento.createWriteRoot("root");
	        IMemento child = mementoWithChild.createChild("child");
	        child.putTextData("text");
	        memento.putMemento(mementoWithChild);
	        IMemento copiedChild = memento.getChild("child");
	        assertEquals("text", copiedChild.getTextData());
	    }



}
