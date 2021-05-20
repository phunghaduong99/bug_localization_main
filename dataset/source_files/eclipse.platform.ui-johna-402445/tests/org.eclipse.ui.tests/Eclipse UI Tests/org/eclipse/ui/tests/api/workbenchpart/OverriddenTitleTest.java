/*******************************************************************************
 * Copyright (c) 2004, 2010 IBM Corporation and others.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     IBM Corporation - initial API and implementation
 *******************************************************************************/
package org.eclipse.ui.tests.api.workbenchpart;

import junit.framework.Assert;

import org.eclipse.ui.IPropertyListener;
import org.eclipse.ui.IWorkbenchPage;
import org.eclipse.ui.IWorkbenchPart2;
import org.eclipse.ui.IWorkbenchPartConstants;
import org.eclipse.ui.IWorkbenchWindow;
import org.eclipse.ui.tests.harness.util.UITestCase;

/**
 * @since 3.0
 */
public class OverriddenTitleTest extends UITestCase {
    /**
     * @param testName
     */
    public OverriddenTitleTest(String testName) {
        super(testName);
    }

    IWorkbenchWindow window;

    IWorkbenchPage page;

    OverriddenTitleView view;

    boolean titleChangeEvent = false;

    boolean nameChangeEvent = false;

    boolean contentChangeEvent = false;

    private IPropertyListener propertyListener = new IPropertyListener() {
        /* (non-Javadoc)
         * @see org.eclipse.ui.IPropertyListener#propertyChanged(java.lang.Object, int)
         */
        public void propertyChanged(Object source, int propId) {
            switch (propId) {
            case IWorkbenchPartConstants.PROP_TITLE:
                titleChangeEvent = true;
                break;
            case IWorkbenchPartConstants.PROP_PART_NAME:
                nameChangeEvent = true;
                break;
            case IWorkbenchPartConstants.PROP_CONTENT_DESCRIPTION:
                contentChangeEvent = true;
                break;
            }
        }
    };

    protected void doSetUp() throws Exception {
        super.doSetUp();
        window = openTestWindow();
        page = window.getActivePage();
        view = (OverriddenTitleView) page
                .showView("org.eclipse.ui.tests.workbenchpart.OverriddenTitleView");
        view.addPropertyListener(propertyListener);
        titleChangeEvent = false;
        nameChangeEvent = false;
        contentChangeEvent = false;
    }

    /* (non-Javadoc)
     * @see org.eclipse.ui.tests.util.UITestCase#doTearDown()
     */
    protected void doTearDown() throws Exception {
        view.removePropertyListener(propertyListener);
        page.hideView(view);
        super.doTearDown();
    }

    private static void verifySettings(IWorkbenchPart2 part,
            String expectedTitle, String expectedPartName,
            String expectedContentDescription) throws Exception {
        Assert.assertEquals("Incorrect view title", expectedTitle, part
                .getTitle());
        Assert.assertEquals("Incorrect part name", expectedPartName, part
                .getPartName());
        Assert.assertEquals("Incorrect content description",
                expectedContentDescription, part.getContentDescription());
    }

    private void verifySettings(String expectedTitle, String expectedPartName,
            String expectedContentDescription) throws Exception {
        verifySettings(view, expectedTitle, expectedPartName,
                expectedContentDescription);
    }

    /**
     * Ensure that we've received the given property change events since the start of the test
     * 
     * @param titleEvent PROP_TITLE
     * @param nameEvent PROP_PART_NAME
     * @param descriptionEvent PROP_CONTENT_DESCRIPTION
     */
    private void verifyEvents(boolean titleEvent, boolean nameEvent,
            boolean descriptionEvent) {
        if (titleEvent) {
            Assert.assertEquals("Missing title change event", titleEvent,
                    titleChangeEvent);
        }
        if (nameEvent) {
            Assert.assertEquals("Missing name change event", nameEvent,
                    nameChangeEvent);
        }
        if (descriptionEvent) {
            Assert.assertEquals("Missing content description event",
                    descriptionEvent, contentChangeEvent);
        }
    }

    public void testDefaults() throws Throwable {
        verifySettings("OverriddenTitle", "OverriddenTitleView",
                "OverriddenTitle");
        verifyEvents(false, false, false);
    }

    public void testCustomName() throws Throwable {
        view.setPartName("CustomPartName");
        verifySettings("OverriddenTitle", "CustomPartName", "OverriddenTitle");
        verifyEvents(false, true, false);
    }

    public void testEmptyContentDescription() throws Throwable {
        view.setContentDescription("");
        verifySettings("OverriddenTitle", "OverriddenTitleView", "");
        verifyEvents(false, false, true);
    }

    public void testCustomTitle() throws Throwable {
        view.customSetTitle("CustomTitle");
        verifySettings("CustomTitle", "OverriddenTitleView", "CustomTitle");
        verifyEvents(true, false, true);
    }

    public void testCustomContentDescription() throws Throwable {
        view.setContentDescription("CustomContentDescription");
        verifySettings("OverriddenTitle", "OverriddenTitleView",
                "CustomContentDescription");
        verifyEvents(false, false, true);
    }

    public void testCustomNameAndContentDescription() throws Throwable {
        view.setPartName("CustomName");
        view.setContentDescription("CustomContentDescription");
        verifySettings("OverriddenTitle", "CustomName",
                "CustomContentDescription");
        verifyEvents(false, true, true);
    }

}
