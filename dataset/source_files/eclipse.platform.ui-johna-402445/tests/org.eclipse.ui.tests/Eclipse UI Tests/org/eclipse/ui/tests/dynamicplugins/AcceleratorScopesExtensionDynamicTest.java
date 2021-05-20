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
package org.eclipse.ui.tests.dynamicplugins;

import org.eclipse.core.commands.common.NamedHandleObject;
import org.eclipse.core.commands.common.NotDefinedException;
import org.eclipse.ui.contexts.IContextService;
import org.eclipse.ui.internal.registry.IWorkbenchRegistryConstants;

/**
 * Tests whether the "org.eclipse.ui.acceleratorScopes" extension point can be
 * added and removed dynamically.
 * 
 * @since 3.1.1
 */
public final class AcceleratorScopesExtensionDynamicTest extends
		DynamicTestCase {

	/**
	 * Constructs a new instance of
	 * <code>AcceleratorScopesExtensionDynamicTest</code>.
	 * 
	 * @param testName
	 *            The name of the test; may be <code>null</code>.
	 */
	public AcceleratorScopesExtensionDynamicTest(final String testName) {
		super(testName);
	}

	/**
	 * Returns the full-qualified identifier of the extension to be tested.
	 * 
	 * @return The extension identifier; never <code>null</code>.
	 */
	protected final String getExtensionId() {
		return "acceleratorScopesExtensionDynamicTest.testDynamicAcceleratorScopeAddition";
	}

	/**
	 * Returns the unqualified identifier of the extension point to be tested.
	 * 
	 * @return The extension point identifier; never <code>null</code>.
	 */
	protected final String getExtensionPoint() {
		return IWorkbenchRegistryConstants.PL_ACCELERATOR_SCOPES;
	}

	/**
	 * Returns the relative location of the folder on disk containing the
	 * plugin.xml file.
	 * 
	 * @return The relative install location; never <code>null</code>.
	 */
	protected final String getInstallLocation() {
		return "data/org.eclipse.acceleratorScopesExtensionDynamicTest";
	}

	/**
	 * Tests whether the items defined in the extension point can be added and
	 * removed dynamically. It tests that the data doesn't exist, and then loads
	 * the extension. It tests that the data then exists, and unloads the
	 * extension. It tests that the data then doesn't exist.
	 */
	public final void testAcceleratorScopes() {
		final IContextService service = (IContextService) getWorkbench()
				.getAdapter(IContextService.class);
		NamedHandleObject namedHandleObject;

		namedHandleObject = service.getContext("monkey");
		try {
			namedHandleObject.getName();
			fail();
		} catch (final NotDefinedException e) {
			assertTrue(true);
		}

		getBundle();

		namedHandleObject = service.getContext("monkey");
		try {
			assertTrue("Monkey".equals(namedHandleObject.getName()));
		} catch (final NotDefinedException e) {
			fail();
		}

		removeBundle();

		namedHandleObject = service.getContext("monkey");
		try {
			namedHandleObject.getName();
			fail();
		} catch (final NotDefinedException e) {
			assertTrue(true);
		}
	}
}
